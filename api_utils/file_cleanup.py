"""
上传文件自动清理模块
"""

import os
import time
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
from config.constants import UPLOAD_FILE_RETENTION_DAYS, FILE_CLEANUP_INTERVAL_HOURS
from server import logger

class FileCleanupService:
    """文件清理服务"""
    
    def __init__(self):
        self.upload_dir = Path(__file__).parent.parent / "upload_images"
        self.is_running = False
        self.cleanup_task = None
        
    def cleanup_expired_files(self):
        """清理过期文件"""
        if not self.upload_dir.exists():
            logger.info("上传目录不存在，跳过清理")
            return 0
        
        current_time = time.time()
        retention_seconds = UPLOAD_FILE_RETENTION_DAYS * 24 * 60 * 60
        deleted_count = 0
        total_size_freed = 0
        
        try:
            for file_path in self.upload_dir.iterdir():
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > retention_seconds:
                        try:
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            deleted_count += 1
                            total_size_freed += file_size
                            logger.info(f"已删除过期文件: {file_path.name} (年龄: {file_age/86400:.1f}天, 大小: {file_size}字节)")
                        except Exception as e:
                            logger.error(f"删除文件失败 {file_path.name}: {e}")
        except Exception as e:
            logger.error(f"清理文件时出错: {e}")
        
        if deleted_count > 0:
            logger.info(f"文件清理完成，删除了 {deleted_count} 个过期文件，释放空间 {total_size_freed/1024/1024:.2f}MB")
        else:
            logger.info("没有找到需要清理的过期文件")
        
        return deleted_count
    
    def get_next_cleanup_time(self):
        """计算下次清理时间（基于配置的间隔时间）"""
        now = datetime.now()
        next_cleanup = now + timedelta(hours=FILE_CLEANUP_INTERVAL_HOURS)
        return next_cleanup
    
    async def start_cleanup_service(self):
        """启动清理服务"""
        if self.is_running:
            logger.warning("文件清理服务已在运行")
            return

        self.is_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

        next_cleanup = self.get_next_cleanup_time()
        logger.info(f"文件清理服务已启动")
        logger.info(f"清理间隔: {FILE_CLEANUP_INTERVAL_HOURS}小时")
        logger.info(f"保留期限: {UPLOAD_FILE_RETENTION_DAYS}天")
        logger.info(f"下次清理时间: {next_cleanup.strftime('%Y-%m-%d %H:%M:%S')}")
    
    async def stop_cleanup_service(self):
        """停止清理服务"""
        if not self.is_running:
            return

        self.is_running = False
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("文件清理服务已停止")
    
    async def _cleanup_loop(self):
        """清理循环 - 按配置的间隔时间执行"""
        while self.is_running:
            try:
                # 等待配置的间隔时间
                wait_seconds = FILE_CLEANUP_INTERVAL_HOURS * 3600
                logger.info(f"等待 {FILE_CLEANUP_INTERVAL_HOURS} 小时后执行下次清理...")

                await asyncio.sleep(wait_seconds)

                if not self.is_running:
                    break

                # 执行清理
                logger.info("开始执行定时文件清理...")
                self.cleanup_expired_files()

            except asyncio.CancelledError:
                logger.info("文件清理循环被取消")
                break
            except Exception as e:
                logger.error(f"文件清理循环出错: {e}")
                # 出错后等待1小时重试
                await asyncio.sleep(3600)

# 全局实例
_cleanup_service = FileCleanupService()

async def start_file_cleanup():
    """启动文件清理"""
    await _cleanup_service.start_cleanup_service()

async def stop_file_cleanup():
    """停止文件清理"""
    await _cleanup_service.stop_cleanup_service()

def manual_cleanup():
    """手动执行文件清理"""
    return _cleanup_service.cleanup_expired_files()

def get_cleanup_status():
    """获取清理服务状态"""
    next_cleanup = _cleanup_service.get_next_cleanup_time()
    return {
        "running": _cleanup_service.is_running,
        "retention_days": UPLOAD_FILE_RETENTION_DAYS,
        "interval_hours": FILE_CLEANUP_INTERVAL_HOURS,
        "next_cleanup_time": next_cleanup.strftime('%Y-%m-%d %H:%M:%S'),
        "upload_dir": str(_cleanup_service.upload_dir)
    }
